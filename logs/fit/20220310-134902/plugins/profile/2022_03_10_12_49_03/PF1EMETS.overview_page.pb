?&$	%㼣j???P?Q?EѤ?0??!??!|
?????$	???=??
@
?????IXWo??@!?VMz?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?1 {?????[='?o??A?`???|??YG8-x?W??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?A?F??????ʆ5???A?m??fc??YSYvQ???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???@,???h?wa??A?s]?@??Y??z?2Q??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails01??c????l?V^????Agd??S??Y?H????rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???̓k??n?HJz??A<?\?gA??Y%??,???rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?r?47??v??y?]??A?8?j?3??Y???$????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0nڌ?U??Ҏ~7???Ast??%??Y?}9?]??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	?~m???????/?^|??AdT8?T??Y???v?Ӓ?rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
 ??????3#????A|?ԗ????Y????c>??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0;ǀ??n?????9??AX?%?????Y?<?E~???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??q?d?????????A~?
Ĳ??Y??B???rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????=m??Ü?M???A?8?Z????Y?!9??U??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails00??!????????A????G??Y?N??o??rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????>???!p$?`??Aa?N"¿??Y??(????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0|
?????ٴR???A+MJA????Y???	/??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?? ?????P?V?f??AO??C??Y?)H4??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??y?)????G?Ȱ??A??^f???Y?hE,??rtrain 97*	+???o?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat??|?X???!?N?Bt?B@):z?ަ???1?[&?F@@:Preprocessing2T
Iterator::Root::ParallelMapV2??????!??6?0@)??????1??6?0@:Preprocessing2E
Iterator::Root?S?D?[??!Ӯ?>N?@)d${??!??1s??iA?-@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::ZipBҧU????!K=Ep,Q@)?l\???1ǿ?X?%@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?ڊ?e???!$?ݚUG$@)?ڊ?e???1$?ݚUG$@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate? ??U??!?XO?h	1@)?,??o???1?8????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*6\?-??!?	??@)6\?-??1?	??@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap???KqU??!?S:,?4@)d??u???1(?W'?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 45.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no97!????
@I??â *X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?L@[[??(?? ???????9??!l?V^????	!       "	!       *	!       2$	??3?np??Kl57ŏ??????G??!+MJA????:	!       B	!       J$	??[ڸ??%y_ڄja?????c>??!%??,???R	!       Z$	??[ڸ??%y_ڄja?????c>??!%??,???b	!       JCPU_ONLYY7!????
@b q??â *X@Y      Y@q?#Ń??T@"?	
both?Your program is POTENTIALLY input-bound because 45.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?83.6% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 