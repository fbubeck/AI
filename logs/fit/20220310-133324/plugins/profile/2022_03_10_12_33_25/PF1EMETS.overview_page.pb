?&$	,?7?ޅ??F????????.???0??!????g???$	?? V?
@??\???kp
???@!L?$LY@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?|	^?????4????Aס??????Y?d??7i??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0S?[?z???'?bd???A?7??w??Y_z?sѐ??rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?.???0??4??X?_??A9a?hV???Y#?	?Y??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?\????e?I)????A????+??Y??4????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0FzQ?_???JbI????AV??????Y?Nx	N}??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0q?GR???????~1??AMg'?????Y|?&???rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0K????????c??A?Ac&Q??YD2??z???rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	??@?ȓ??&?`6??A???M???Y{?l??rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
v??y?]?????????AT??????Y???B??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?iQ??#?k$	??A??M~?N??Yf?s~????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????g?????u?+.??A\?d8????Y-??VД?rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?j?j?R???c?? w??A`vOj??Y??*3????rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?3????????#???A$?jf-??YGW#???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0/?HM;???????>??A:???????Y????^???rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??e?-???`TR'???A??p?Ws??Y???????rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?~?:pN??B]¡??A?(&o????Y?8ӄ?'??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????? v???AK?.??"??Yg?R@????rtrain 97*	Zd;?OK?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat?L?????!a8~?
?B@)???u?|??1K'&??@@:Preprocessing2T
Iterator::Root::ParallelMapV2b???u??!}?6 ?1@)b???u??1}?6 ?1@:Preprocessing2E
Iterator::Root[???i??!??B??@@)S?K?^??1???N 0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip??k?)??!??޷?P@)?jI??1???q_p'@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceʥ??$??!??? #?"@)ʥ??$??1??? #?"@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*????????!???rq@)????????1???rq@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??N???!)?e?^,@)C p??s??1 ?;r@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMapa?unڌ??!?X'???1@)????1v??1%?we?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 43.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9x?????
@I??9?2)X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?,?????? <.???? v???!JbI????	!       "	!       *	!       2$	eWU*????	u??
	??9a?hV???!?Ac&Q??:	!       B	!       J$	??? ?ђ??P?;?ze??Nx	N}??!?d??7i??R	!       Z$	??? ?ђ??P?;?ze??Nx	N}??!?d??7i??b	!       JCPU_ONLYYx?????
@b q??9?2)X@Y      Y@q??L3U?U@"?	
both?Your program is POTENTIALLY input-bound because 43.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?86.0% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 