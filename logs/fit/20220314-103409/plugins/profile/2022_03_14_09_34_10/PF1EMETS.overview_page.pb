?&$	>???? ??V?????6?^?7??!???B??$	????b	@????????L?????@!G?#قp@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????|???W?'???A???????Y ?Z? m??rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???B???g??????Ad?6??:??Y?@.q䁘?rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$	?P(???????A??!???Y???~??rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0.7??B?????쟧??Auۈ'???Y?s???)??rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?6?^?7??? ?S?D??A{???w???Yx?'-\V??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??g?z??(~??k	??AfM,????Y?<c_????rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0e??E
?????????A???-??Y????%ƒ?rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	C???|?????????A??{ds???Y9?M?a???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
?>??s????B?y???A?? =E??YOyt#,*??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0c	kc????겘?|\??A?mQf?L??Y??Co???rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?Ǚ&l????uʣ??A?)?"??Y? ????rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???4`???L?T?#??A?0~????Yn????^??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0$?\???(?bd???A^??N??Y'L?????rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0}^??#??B??	ܺ??A??P???Y?????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?vۅ????.???A=?+J	???Y???&M???rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails08?a?A
??????????A+l? [??Y????o{??rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0E?$]3y???҆?????A?t ?????Y??U?6œ?rtrain 97*	5^?I?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatt?^????!???O?AB@)#??^??1&?qDc?@@:Preprocessing2E
Iterator::Root-??m??!P?P??@@)y?&1???1/??(=?0@:Preprocessing2T
Iterator::Root::ParallelMapV2??
?.??!q~?xZ0@)??
?.??1q~?xZ0@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip%?}?e???!X??ׯ?P@)?n??o???1????&@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceDL?$z??!?8?3??#@)DL?$z??1?8?3??#@:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate;?zj???!?t 3?-@)b??????14x???
@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap????G??!sQ&<(3@)?~j?t???1?[X???@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?D?
)??!??.??@)?D?
)??1??.??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 42.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????`V	@Ib+?L5X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	W??x_C??	`?*?̀?? ?S?D??!.???	!       "	!       *	!       2$	pXb??????cA!??{???w???!?mQf?L??:	!       B	!       J$	??s?_B????6??e???Co???! ?Z? m??R	!       Z$	??s?_B????6??e???Co???! ?Z? m??b	!       JCPU_ONLYY????`V	@b qb+?L5X@Y      Y@q???F8S@"?	
both?Your program is POTENTIALLY input-bound because 42.8% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?76.9% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 