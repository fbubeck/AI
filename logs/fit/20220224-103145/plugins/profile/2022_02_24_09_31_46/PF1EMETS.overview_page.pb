?&$	b??h????1?<ض?*?J=B??!???????$	???#{1@|??d ???Lۙgp@!?4^?b?@"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???????b1?Z{???A=?E~???Y??????rtrain 81"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails05??a0???????3???A??jׄ???Y?BB???rtrain 82"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0׿????*q????A??5???Y#/kb????rtrain 83"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0??Y?????)????A??RB????Y)? ???rtrain 84"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0+???????D.8???Ay???????Y?o?[??rtrain 85"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?|?5^:?????|y??A5?b??^??YEeÚʢ??rtrain 86"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0n/????4g}?1Y??A?Z|
????YW#??2??rtrain 87"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0	8??@??6?.6???A??0????Y???I???rtrain 88"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
*?J=B??J_9???A?vLݕ]??Y???0??rtrain 89"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0?4'/2????ND????Ar???	??YD? ????rtrain 90"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0]????[??N??ĭ??A?~k'JB??Yz9??cx??rtrain 91"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0ٵ?ݒ??_ѭ?????AW??Ma???Y??N]?,??rtrain 92"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0? Ϡ!???I?>???A35	ސF??Y=((E+???rtrain 93"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0????????v?>X??A*?:]???YzUg????rtrain 94"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0???=???y>?ͨ??A"U?????Y+???ڧ??rtrain 95"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0
,?)??X8I?Ǵ??A?TގpZ??Y}??????rtrain 96"q
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails0D??<?????UJ????A?!q????Y?3?<F??rtrain 97*	/?$?I?@2k
4Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeatQg?!?{??!???1??@)?^f?(??1;3EO?O=@:Preprocessing2E
Iterator::Root-??;????!=`UrH?D@)"3?<???1??%%?:@:Preprocessing2T
Iterator::Root::ParallelMapV2m??~????!?f
???,@)m??~????1?f
???,@:Preprocessing2Y
"Iterator::Root::ParallelMapV2::Zip(,???)??!ß???PM@)ı.n???1j+c%h$@:Preprocessing2?
NIterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceN{JΉ=??!???DS @)N{JΉ=??1???DS @:Preprocessing2u
>Iterator::Root::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??E?T???!ȯsm??)@)?D?A???1?5??	@:Preprocessing2e
.Iterator::Root::ParallelMapV2::Zip[0]::FlatMap??(????!?9?0@){JΉ=???1????@:Preprocessing2w
@Iterator::Root::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*6#??E???!??'?@)6#??E???1??'?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???K?@IR%C;`?W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	SY?l???s?v??ܲ???v?>X??!????3???	!       "	!       *	!       2$	;4'*?????i@?͢??vLݕ]??!35	ސF??:	!       B	!       J$	???Qw???g?j??v??=((E+???!#/kb????R	!       Z$	???Qw???g?j??v??=((E+???!#/kb????b	!       JCPU_ONLYY???K?@b qR%C;`?W@Y      Y@q??s?f?S@"?	
both?Your program is POTENTIALLY input-bound because 50.1% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQb?78.7% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 